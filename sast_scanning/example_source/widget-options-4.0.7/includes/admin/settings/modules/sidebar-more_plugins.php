<?php
/**
 * Support Sidebar Metabox
 * Settings > Widget Options
 *
 * @copyright   Copyright (c) 2016, Jeffrey Carandang
 * @since       4.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Metabox for Support
 *
 * @since 4.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_more_plugins' ) ):
	function widgetopts_settings_more_plugins(){ ?>
		<div id="widgetopts-sidebar-widget-more_plugins" class="postbox widgetopts-sidebar-widget">
			<h3 class="hndle ui-sortable-handle"><span><?php _e( 'Ready for Gutenberg?', 'widget-options' );?></span></h3>
			<div class="inside">
				<p><?php _e( 'Get the same controls for your Gutenberg blocks using our brand new plugin Block Options for Gutenberg. Also available for free on plugin repository.', 'widget-options' );?></p>
				<p>
					<a class="button-secondary" href="https://wordpress.org/plugins/block-options/" target="_blank"><?php _e( 'Download Block Options', 'widget-options' );?></a>
				</p>
			</div>
		</div>

	    <?php
	}
	//add_action( 'widgetopts_module_sidebar', 'widgetopts_settings_more_plugins', 25 );
endif;
?>
