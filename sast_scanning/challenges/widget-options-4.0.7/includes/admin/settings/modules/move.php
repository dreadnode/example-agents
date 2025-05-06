<?php
/**
 * Move Widgets Module
 * Settings > Widget Options :: Move Widget
 *
 * @copyright   Copyright (c) 2017, Jeffrey Carandang
 * @since       3.4
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) exit;

/**
 * Create Card Module for Move Widget Feature
 *
 * @since 3.4
 * @global $widget_options
 * @return void
 */

  /*
 * Note: Please add a class "no-settings" in the <li> card if the card has no additional configuration, if there are configuration please remove the class
 */

if( !function_exists( 'widgetopts_settings_move' ) ):
    function widgetopts_settings_move(){
        global $widget_options;

        //avoid issue after update
        if( !isset( $widget_options['move'] ) ){
            $widget_options['move'] = '';
        }
        ?>
        <li class="widgetopts-module-card widgetopts-module-card-no-settings no-settings <?php echo ( $widget_options['move'] == 'activate' ) ? 'widgetopts-module-type-enabled' : 'widgetopts-module-type-disabled'; ?>" id="widgetopts-module-card-move" data-module-id="move">
    		<div class="widgetopts-module-card-content">
    			<h2><?php _e( 'Move Widget', 'widget-options' );?></h2>
    			<p class="widgetopts-module-desc">
    				<?php _e( 'Easily move widgets to any sidebar widget area without drag and drop.', 'widget-options' );?>
    			</p>

    			<div class="widgetopts-module-actions hide-if-no-js">
                    <?php if( $widget_options['move'] == 'activate' ){ ?>
    					<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
    					<button class="button button-secondary widgetopts-toggle-activation"><?php _e( 'Disable', 'widget-options' );?></button>
    				<?php }else{ ?>
    					<button class="button button-secondary widgetopts-toggle-settings"><?php _e( 'Learn More', 'widget-options' );?></button>
    					<button class="button button-primary widgetopts-toggle-activation"><?php _e( 'Enable', 'widget-options' );?></button>
    				<?php } ?>

    			</div>

    		</div>

    		<?php widgetopts_modal_start( $widget_options['move'] ); ?>
    			<span class="dashicons widgetopts-dashicons widgetopts-no-top dashicons-image-rotate-right"></span>
    			<h3 class="widgetopts-modal-header"><?php _e( 'Move Widget', 'widget-options' );?></h3>
    			<p>
    				<?php _e( 'Move Widget feature will automatically add <strong>Move</strong> button that will let you easily move any widgets to any sidebar widget areas without dragging them. This will definitely increase your productivity and widget management specially on smaller screen devices such as mobile phones. You can check how this feature works on this <a href="https://vimeo.com/229112330" target="_blank">video</a>. Thanks!', 'widget-options' );?>
    			</p>
    			<p class="widgetopts-settings-section">
    				<?php _e( 'No additional settings available.', 'widget-options' );?>
    			</p>
    		<?php widgetopts_modal_end( $widget_options['move'] ); ?>

    	</li>
        <?php
    }
    add_action( 'widgetopts_module_cards', 'widgetopts_settings_move', 64 );
endif;
?>
