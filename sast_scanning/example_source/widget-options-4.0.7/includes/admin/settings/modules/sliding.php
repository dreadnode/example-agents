<?php
/**
 * Widgets on Menu Settings Module
 * Settings > Widget Options :: Widgets on Menu
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       1.0
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Widgets on Menu Options
 *
 * @since 1.0
 * @return void
 */
if( !function_exists( 'widgetopts_settings_sliding_addon' ) ):
	function widgetopts_settings_sliding_addon(){

		if( !is_plugin_active( 'sliding-widget-options/plugin.php' ) ){ ?>
	        <li class="widgetopts-module-card widgetopts-module-type-pro" data-module-id="sliding">
				<div class="widgetopts-module-card-content" style="border-color: #ffb310;">
					<a href="<?php echo apply_filters('widget_options_site_url', trailingslashit(WIDGETOPTS_PLUGIN_WEBSITE).'modal-pop-up-and-sliding-widget-options/?utm_source=wordpressadmin&utm_campaign=modulecard');?>" target="_blank" class="widgetopts-pro-upsell"></a>
					<h2><?php _e( 'Sliding Widgets Add-on', 'widget-options' );?></h2>
					<div class="widgetopts-pro-label"><span class="dashicons dashicons-admin-plugins"></span></div>
					<p class="widgetopts-module-desc">
						<?php
						_e( 'Premium add-on to transform any widgets into Modal Pop-up, Slide-ins and Sliding Panels.', 'widget-options' );
						if( defined( 'ELEMENTOR_VERSION' ) ){
							_e( '<br /><span class="dashicons dashicons-yes" style="color: #ffb310;"></span> Elementor Compatible', 'widget-options' );
						}
						?>
					</p>
				</div>
			</li>
	    <?php
		}
	}
	add_action( 'widgetopts_module_cards', 'widgetopts_settings_sliding_addon', 220 );
endif;
?>
